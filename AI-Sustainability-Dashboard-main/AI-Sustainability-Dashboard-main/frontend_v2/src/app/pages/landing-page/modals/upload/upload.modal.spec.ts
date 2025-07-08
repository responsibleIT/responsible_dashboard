import { ComponentFixture, TestBed } from '@angular/core/testing';

import { UploadModal } from './upload.modal';

describe('UploadComponent', () => {
  let component: UploadModal;
  let fixture: ComponentFixture<UploadModal>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [UploadModal]
    })
    .compileComponents();

    fixture = TestBed.createComponent(UploadModal);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
