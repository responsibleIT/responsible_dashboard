import { ComponentFixture, TestBed } from '@angular/core/testing';

import { BenchmarkLoaderComponent } from './benchmark-loader.component';

describe('ValidationLoaderComponent', () => {
  let component: BenchmarkLoaderComponent;
  let fixture: ComponentFixture<BenchmarkLoaderComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [BenchmarkLoaderComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(BenchmarkLoaderComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
